{ pkgs, pyrl }:
let
  dev = pkgs.buildEnv {
    name = "pyrl-dev";
    paths = [
      pkgs.python-language-server
      (pkgs.python3.withPackages (ps: with ps; [
        pyrl
        pyls-black
        ipdb
        rope
        pyflakes
        pytest
        tensorflow-tensorboard_2
      ]))
    ];
  };
in
pkgs.mkShell {
  buildInputs = pyrl.pkg.propagatedBuildInputs ++ [
    pyrl.dev
  ];
  shellHook = ''
    export PYTHONPATH=$PYTHONPATH:$PWD
  '';
}
