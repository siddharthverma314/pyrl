{ stdenv
, buildPythonPackage
, fetchurl
, isPy37
, isPy38
, python
, linuxPackages
, addOpenGLRunpath
, future
, numpy
, patchelf
, pyyaml
, requests
, typing-extensions
, fetchPypi
}:

buildPythonPackage {
  pname = "pytorch";
  version = "1.7.0";

  format = "wheel";

  disabled = !(isPy37 || isPy38);

  src = fetchurl {
    name = "torch-1.7.0+cu110-cp38-cp38-linux_x86_64.whl";
    url = "https://download.pytorch.org/whl/cu110/torch-1.7.0%2Bcu110-cp38-cp38-linux_x86_64.whl";
    sha256 = "sha256-VF/jhXrcGarJJGnJW871tc9DKVAKQhTfTtUYkLbhK8E=";
  };

  nativeBuildInputs = [
    addOpenGLRunpath
    patchelf
  ];

  propagatedBuildInputs = [
    future
    numpy
    pyyaml
    requests
    typing-extensions
  ];

  # PyTorch are broken: the dataclasses wheel is required, but ships with
  # Python >= 3.7. Our dataclasses derivation is incompatible with >= 3.7.
  #
  # https://github.com/pytorch/pytorch/issues/46930
  #
  # Should be removed with the next PyTorch version.
  pipInstallFlags = [
    "--no-deps"
  ];

  postInstall = ''
    # ONNX conversion
    rm -rf $out/bin
  '';

  postFixup = let
    rpath = stdenv.lib.makeLibraryPath [ stdenv.cc.cc.lib linuxPackages.nvidia_x11 ];
  in ''
    find $out/${python.sitePackages}/torch/lib -type f \( -name '*.so' -or -name '*.so.*' \) | while read lib; do
      echo "setting rpath for $lib..."
      patchelf --set-rpath "${rpath}:$out/${python.sitePackages}/torch/lib" "$lib"
      addOpenGLRunpath "$lib"
    done
  '';

  pythonImportsCheck = [ "torch" ];
}
