JOBS=4  # number of parallel threads

default:
	cachix use pyrl
	nocachix

nocachix:
	nix build -f default.nix pkg dev --max-jobs=$(JOBS)

pushcachix:
	nix-build default.nix | cachix push pyrl -j=$(JOBS)
