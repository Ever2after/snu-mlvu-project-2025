def concat(input_path1, input_path2, output_path, direction='row'):
    from moviepy import VideoFileClip, clips_array
    import time

    start = time.time()

    clip1 = VideoFileClip(input_path1)
    clip2 = VideoFileClip(input_path2)

    assert clip1.fps == clip2.fps, f"FPS 불일치: {clip1.fps} vs {clip2.fps}"
    assert clip1.size == clip2.size, f"해상도 불일치: {clip1.size} vs {clip2.size}"

    if direction == 'row':
        final_clip = clips_array([[clip1, clip2]]) # row-wise
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            fps=clip1.fps,
            audio_codec="aac"
    )
    else:
        final_clip = clips_array([[clip2], [clip1]]) # column-wise
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            fps=clip1.fps,
            audio_codec="aac"
        )
    end = time.time()
    print(f"Concatenation completed in {end - start:.2f} seconds.")