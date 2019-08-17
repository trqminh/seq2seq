class BatchGenerator(object):
    def __init__(self, text, batch_size, sequence_length):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._sequence_length = sequence_length
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]

    def next(self):
        """Generate the next array of batches from the data.
        """
        batches = [None] * self._batch_size
        for i in range(self._batch_size):
            batches[i] = self._text[self._cursor[i]:self._cursor[i]+self._sequence_length].ljust(self._sequence_length)
            self._cursor[i] = (self._cursor[i] + self._sequence_length - 1) % self._text_size
        return batches

    def get_seq_len(self):
        return self._sequence_length
