{ mjKeyPath
, cudaSupport ? false
, fetchFromGitHub
, mesa
, python3
, libGL
, gcc
, stdenv
, callPackage
, autoPatchelfHook
, xorg
, lib
, libglvnd
}:
let
  mach-nix = import (fetchFromGitHub {
    owner = "DavHau";
    repo = "mach-nix";
    rev = "f11fe2b5cb1cb2eae7074c7c52025ff09b1999a3";
    sha256 = "1jg9gdxw7xwpnhh2wg2qcvzhmhckzpkfbplw7q6vma55jwhgkf64";
  });
  mujoco = (callPackage ./mujoco.nix {});
  src = fetchFromGitHub {
    owner = "siddharthverma314";
    repo = "mujoco-py";
    rev = "84f0279f9a78cf1f36f47ac9f8871042b59846a";
    sha256 = "18z19qmmxd83knhfw9df4pjkp0pzjlpbmazx0w82g2ni618vcgry";
  };
in
mach-nix.buildPythonPackage {
  inherit src;
  pname = "mujoco-py";
  version = "1.50.1.1";
  requirements = builtins.readFile "${src}/requirements.txt";

  python = python3;
  MUJOCO_BUILD_GPU = cudaSupport;
  nativeBuildInputs = [ autoPatchelfHook ];
  buildInputs = [
    mesa
    mesa.osmesa
    mujoco
    python3
    libGL
    gcc
    stdenv.cc.cc.lib
  ] ++ lib.optionals cudaSupport [ xorg.libX11 libglvnd ];

  # hacks to make the package work
  postInstall = ''
    cat ${mjKeyPath} > $out/lib/${python3.libPrefix}/site-packages/mujoco_py/mjkey.txt
  '' + lib.optionalString cudaSupport ''
    patchelf --add-needed libEGL.so $out/lib/${python3.libPrefix}/site-packages/mujoco_py/cymj.cpython*.so
    patchelf --add-needed libOpenGL.so $out/lib/${python3.libPrefix}/site-packages/mujoco_py/cymj.cpython*.so
  '';
  doCheck = false;
}
