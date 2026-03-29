from src.cvi.api.video_preview import is_browser_compatible_mp4, preview_url_for_analysis


def test_is_browser_compatible_mp4_accepts_h264_yuv420_aac():
    assert is_browser_compatible_mp4("h264", "yuv420p", "aac") is True


def test_is_browser_compatible_mp4_rejects_mpeg4_video():
    assert is_browser_compatible_mp4("mpeg4", "yuv420p", "aac") is False


def test_is_browser_compatible_mp4_rejects_non_420_pixel_format():
    assert is_browser_compatible_mp4("h264", "yuv422p", "aac") is False


def test_preview_url_for_analysis_is_stable():
    assert preview_url_for_analysis("analysis-123") == "/preview/analysis-123"
