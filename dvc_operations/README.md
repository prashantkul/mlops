## How To Use DVC

### Suggested Workflow

1. run `upload.sh` to upload the data to DVC
2. run `push_new_records.sh` to push the new records to Git

### Uploading Files to DVC

To upload a file to DVC, run the following command:

```bash
./upload.sh <file_to_upload>
```

Replace `<file_to_upload>` with the path to the file you want to upload.

#### Pushing New Records to DVC
DVC records get created when you run the `upload.sh` script. To push new records to Git, run the following command:

```bash
./push_new_records.sh
```

### Downloading Files from DVC

#### Reading a file to a pandas DataFrame

To read a file from DVC to a pandas DataFrame, run the following command:

```bash
./read.py <file_path>
```

Replace `<file_path>` with the path to the file you want to read.

#### Saving a dvc file to a CSV file

To save a pandas DataFrame to a CSV file, run the following command:

```bash
./save.py <file_path>
```

Replace `<file_path>` with the path to the file you want to save.
