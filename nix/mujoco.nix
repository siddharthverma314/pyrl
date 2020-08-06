{ stdenv
, fetchurl
, autoPatchelfHook
, unzip
, libGL
, xorg
}:
stdenv.mkDerivation {
  pname = "mujoco";
  version = "1.5";
  src = fetchurl {
    url = "https://www.roboti.us/download/mjpro150_linux.zip";
    sha256 = "0xsxng45q27fr25m018jci0f3axv6h9y4zwxj62w4apbjrcyh9pv";
  };
  buildInputs = [
    autoPatchelfHook
    unzip
    stdenv.cc.cc.lib
    libGL
    xorg.libX11
    xorg.libXinerama
    xorg.libXxf86vm
    xorg.libXcursor
    xorg.libXrandr
  ];
  installPhase = ''
    mkdir $out

    # copy required folders
    for folder in bin include model; do
      cp -r $folder $out/$folder
    done

    # make lib folder
    mkdir $out/lib
    ln -s $out/bin/*.so $out/lib/
  '';
  testPhase = ''
    cd sample
    make
  '';
}
