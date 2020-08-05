{ callPackage
, buildPythonPackage
, fetchPypi
, numpy
, cython
}:
buildPythonPackage rec {
  pname = "cpprb";
  version = "9.2.0";
  src = fetchPypi {
    inherit pname version;
    sha256 = "001164hd2abn9gkzgsc2gw4mb4rwd6vxbg5ymbg4pf99dm4vswba";
  };
  propagatedBuildInputs = [ numpy cython ];
}
