{ callPackage
, buildPythonPackage
, fetchPypi
, six
, pathlib2
, setuptools
, sources
}:
buildPythonPackage rec {
  pname = "flatten-dict";
  version = "0.3.0";
  src = sources.flatten-dict;
  postPatch = ''
    cat > setup.py << EOF
    from setuptools import setup, find_packages

    setup(
      name='flatten-dict',
      version='0.0.0',
      author='Siddharth Verma',
      packages=find_packages(),    
    )
    EOF
  '';
  propagatedBuildInputs = [ six pathlib2 setuptools ];
}
