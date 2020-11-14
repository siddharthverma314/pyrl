{ cudaSupport ? true
, python ? "python38"
, mjKeyPath ? ~/secrets/mjkey.txt
, inputs
}:
final: prev: {
  # add sources
  sources = inputs;

  # top-level pkgs overlays
  ffmpeg = prev.ffmpeg.override {
    nonfreeLicensing = true;
    nvenc = cudaSupport; # nvidia support
  };

  gitignore = (prev.callPackage inputs.gitignore {}).gitignoreSource;

  pythonOverrides = python-self: python-super: rec {
    blas = prev.blas.override { blasProvider = prev.mkl; };
    lapack = prev.lapack.override { lapackProvider = prev.mkl; };

    pytorch = python-super.pytorch.override {
      inherit cudaSupport;
      tensorflow-tensorboard = python-super.tensorflow-tensorboard_2;
    };

    opencv3 = python-super.opencv3.override {
      enableCuda = cudaSupport;
      enableFfmpeg = true;
    };

    opencv4 = python-super.opencv4.override {
      enableCuda = cudaSupport;
      enableFfmpeg = true;
    };

    mujoco-py = python-super.callPackage ./mujoco_py.nix {
      inherit cudaSupport mjKeyPath;
      mesa = prev.mesa;
    };

    cpprb = python-super.callPackage ./cpprb.nix {};

    cloudpickle = python-super.callPackage ./cloudpickle.nix {};

    gym = python-super.gym.overrideAttrs (old: {
      postPatch = ''
          substituteInPlace setup.py \
            --replace "pyglet>=1.2.0,<=1.3.2" "pyglet" \
        '';
    });

    # self-made packages
    glfw = python-super.callPackage ./glfw.nix {};
    flatten-dict = python-super.callPackage ./flatten-dict.nix {};
    scikit-video = python-super.callPackage ./scikit-video.nix {};
  };

  "${python}" = prev."${python}".override {
    packageOverrides = final.pythonOverrides;
  };
  python3 = final."${python}";
}
