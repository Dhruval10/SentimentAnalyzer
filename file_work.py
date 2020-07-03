"""This class will work on writing and reading from files."""

class FileOperations():

    def __init__(self, file_name):
        self._file_name=file_name

    def write(self, text_to_write):
        """To open specified file in write mode.

        Args:
            text_to_write: write provided list to file.
        """
        file = open(self._file_name, 'w')
        for test_string in text_to_write:
            file.write(test_string+'\n')
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
