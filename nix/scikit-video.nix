{ buildPythonPackage
, fetchFromGitHub
, numpy
, scipy
, scikitlearn
, pillow
, ffmpeg
, nose
}:
buildPythonPackage rec {
  pname = "scikit-video";
  version = "1.1.11";
  src = fetchFromGitHub {
    owner = "scikit-video";
    repo = "scikit-video";
    rev = "87c7113a84b50679d9853ba81ba34b557f516b05";
    sha256 = "1vaqkw7vag012qqbfmhbnpkb153mxa30qz74l5b490rzlw5lq13g";
  };
  postPatch = ''
    substituteInPlace skvideo/__init__.py \
      --replace '_FFMPEG_PATH = which("ffprobe")' '_FFMPEG_PATH = "${ffmpeg}/bin"'
  '';
  propagatedBuildInputs = [
    # python inputs
    numpy
    scipy
    scikitlearn
    pillow

    # non-python inputs
    ffmpeg
  ];
  checkInputs = [
    nose
  ];
}
