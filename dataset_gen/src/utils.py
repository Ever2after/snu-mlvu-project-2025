def concat(input_path1, input_path2, output_path1, output_path2):
    from moviepy import VideoFileClip, clips_array

    clip1 = VideoFileClip(input_path1)
    clip2 = VideoFileClip(input_path2)

    assert clip1.fps == clip2.fps, f"FPS 불일치: {clip1.fps} vs {clip2.fps}"
    assert clip1.size == clip2.size, f"해상도 불일치: {clip1.size} vs {clip2.size}"

    final_clip1 = clips_array([[clip1, clip2]]) # row-wise
    final_clip2 = clips_array([[clip2], [clip1]]) # column-wise

    final_clip1.write_videofile(
        output_path1,
        codec="libx264",
        fps=clip1.fps,
        audio_codec="aac",
        threads=16,
    )
    final_clip2.write_videofile(
        output_path2,
        codec="libx264",
        fps=clip1.fps,
        audio_codec="aac",
        threads=16,
    )