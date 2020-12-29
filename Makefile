JOBS=4  # number of parallel threads

default:
	cachix use pyrl
	make nocachix
	make pushcachix

nocachix:
	nix build --max-jobs $(JOBS)

pushcachix:
	nix path-info | cachix push pyrl -j $(JOBS)
