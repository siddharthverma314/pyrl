{ callPackage
, buildPythonPackage
, fetchPypi
, glfw3
, python3
}:
buildPythonPackage rec {
  pname = "glfw";
  version = "1.12.0";
  src = fetchPypi {
    inherit pname version;
    sha256 = "1ccgpq555p0ay6gp9lcbm3yp1d56yfcnjgch0cb4ypj78dxfv5gi";
  };
  preFixup = ''
    cat <<EOF > $out/lib/${python3.libPrefix}/site-packages/glfw/library.py
    import ctypes
    glfw = ctypes.CDLL("${glfw3}/lib/libglfw.so")
    EOF
    echo FUCK OUT
    cat $out/lib/${python3.libPrefix}/site-packages/glfw/library.py
  '';
  buildInputs = [ glfw3 ];
  doCheck = false;
}
