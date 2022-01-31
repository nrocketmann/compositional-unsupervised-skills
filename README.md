# Workflow

To run experiments, create an instance of the `Experimenter` class
and pass it the arguments it needs.

After every experiment, be sure to run `copy_logs.sh` to get all the metadata back from the server.

If PyCharm is every failing to sync with the server, you can run `copy.sh` from the `empowerment` directory
to sync all files. Don't do this before running `copy_logs.sh` though.
