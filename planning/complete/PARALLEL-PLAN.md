# Parallelization Plan & Bottleneck Analysis

Looking at your Task Manager screenshot, we can clearly see what's happening:

*   **GPU Video Encode is at ~2%**
*   **GPU 3D is at ~5-15%**
*   **CPU is at 15%** (which likely means 1 or 2 cores are maxed out at 100%, while the rest do nothing)
*   **Memory has plenty of headroom** (41 / 128 GB)

**Conclusion:** The bottleneck is **not** the GPU. The NVENC encoder is basically asleep, waiting for frames. The pipeline is **CPU-bound** on single-thread performance. Specifically, the on-demand `Pillow.resize(Image.LANCZOS)` and NumPy slicing in `ken_burns_at_t()` are maxing out a single CPU core. 

Yes, parallelizing frame generation is a **great idea**. Here are three ways to do it, ordered from easiest to most complex.

---

## Option 1: MoviePy Native Threading (The "Quick Win")

MoviePy's `write_videofile` accepts a `threads` argument. By default, it generates one frame, passes it to FFmpeg, generates the next, etc. By increasing `threads`, MoviePy uses background workers to compute upcoming frames concurrently.

**How to implement:**
Modify `slideshow_builder.py` around line 680:
```python
video.write_videofile(
    str(output_path),
    fps=settings.video_fps,
    codec=codec,
    audio_codec="aac",
    bitrate="15000k",
    preset="fast",
    threads=8,  # <--- ADD THIS
    logger="bar"
)
```
*   **Pros:** 1-line change. Pillow's `resize` operation actually releases the Python GIL (Global Interpreter Lock), so multithreading works well here.
*   **Cons:** MoviePy's internal threading can sometimes be buggy with complex composites (like crossfades), but usually works fine.

---

## Option 2: Pre-fetching Frame Buffer (The "Producer-Consumer")

If Option 1 is unstable or doesn't yield enough speedup, we can decouple frame generation from MoviePy's render loop using a queue.

We would wrap the final `video` clip in a custom frame generator that spawns a `concurrent.futures.ThreadPoolExecutor`. 
*   The workers compute frames `t+1`, `t+2`... `t+16` in parallel.
*   They place the computed NumPy arrays into a bounded queue (e.g., max size 32 frames, taking ~200MB RAM).
*   MoviePy simply pulls the already-computed frames from the queue.

*   **Pros:** Keeps memory low, guarantees the GPU encoder is constantly fed.
*   **Cons:** Requires writing a custom wrapper around MoviePy's `Clip.iter_frames()`.

---

## Option 3: Pure GPU Ken Burns (The "Nuclear Option")

Right now, we pull the image into CPU RAM, do math on the CPU, crop on the CPU, resize on the CPU, and then send it to the GPU just to encode. 

Instead of generating frames in Python, we could rewrite `build_photo_clips` to generate **FFmpeg filter graphs** (specifically the `zoompan` filter). FFmpeg would handle the Ken Burns effect entirely in native code (and optionally hardware).

*   **Pros:** Incredibly fast. Zero Python overhead during render.
*   **Cons:** Re-writing the entire core logic of the app. FFmpeg's `zoompan` filter is notoriously finicky, and aligning it with crossfades and audio tracks requires complex filter strings.

---

## Recommendation

**I recommend we start with Option 1 immediately.** It's a single parameter change. Pillow is highly optimized in C, and throwing 8 threads at it should immediately feed the GPU faster and drop your render times significantly.

If Option 1 crashes MoviePy or doesn't speed things up, we move to **Option 2**. We have plenty of RAM (87 GB free!), so holding 16-32 high-res frames in memory at once is perfectly safe. 

Would you like me to implement Option 1 (adding the `threads` argument) so you can test the speedup on your next run?
