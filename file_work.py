"""This class will work on writing and reading from files."""

class FileOperations():

    def __init__(self, file_name):
        self._file_name=file_name

    def write(self, text_to_write):
        """To open specified file in write mode.

        Args:
            text_to_write: write provided list to file.
        """
        file = open(self._file_name, 'a')
        for test_string in text_to_write:
            file.writelines(test_string.lower()+'\n')
        file.close()

    def read(self):
        """Reads all from file.

        Returns:
            A list containing all the lines from file.
        """
        file = open(self._file_name, 'r')
        tweets = file.read().lower().split('\n')
        file.close()
        return tweets

    def remove_blines(self):
        """
        Remove all blank lines from file

        Return:
        """
        with open(self._file_name) as in_file, open(self._file_name, 'r+') as out_file:
            out_file.writelines(line for line in in_file if line.strip())
            out_file.truncate()
