{ callPackage
, buildPythonPackage
, fetchPypi
, six
, pathlib2
}:
buildPythonPackage rec {
  pname = "flatten-dict";
  version = "0.3.0";
  src = fetchPypi {
    inherit pname version;
    sha256 = "0zix664j0n6fyk5infiif38k56knyra95lbs73pwb13wbkql7k0c";
  };
  propagatedBuildInputs = [ six pathlib2 ];
}
