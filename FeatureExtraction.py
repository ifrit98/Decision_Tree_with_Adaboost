class FeatureData:
    """
    Data-set object that contains all relevant training information for model
    """
    def __init__(self, examples, features=None, label=-1, values=None, weight=-2):

        self.examples = examples
        self.weight = weight
        self.label = label

        if not values:
            self.values = list(map(self.unique, zip(*self.examples)))

        if self.examples and not features:
            self.features = list(range(len(self.examples[0])))

        self.inputs = [a for a in self.features if a != self.label]

    def unique(self, seq):
        return list(set(seq))

class FeatureParser:
    """
    Parser object that contains common words to English and Dutch,
    along with special characters to test on.
    """
    __slots__ = 'DutchWords','EnglishWords','SpecialChars','features'

    def __init__(self,fn, test=False):
        self.DutchWords = ['zijn','mijn','voor','jij','hebben']
        self.EnglishWords = ['have','for','his','my']
        self.SpecialChars = ['á', 'é', 'í', 'ó', 'ú', 'à','ä', 'è', 'ë', 'ï', 'ö','ð',
                        'ü','ý','ÿ','Á', 'É', 'Í', 'Ó', 'Ú','Ä', 'À', 'È', 'Ë','Ï', 'Ö', 'Ü','Ÿ',
                            'ö','ë','š','ě','á','ĳ','í','ó','ú']
        self.features = self.parser(fn, test)

    def specialChars(self, exampleChars):
        # Look for special characters
        for chr in exampleChars:
            if ord(chr) > 127:
                return True
        return False

    def nlWords(self,lineTokens):
        # Look for common Dutch words
        for word in lineTokens:
            for nlWord in self.DutchWords:
                if word == nlWord:
                    return True
        return False

    def enWords(self, lineTokens):
        # Look for common English words
        for word in lineTokens:
            for enWord in self.EnglishWords:
                if word == enWord:
                    return True
        return False

    def parser(self,fn, test=False):
        """
        Assumes data is in format:
            'en | Sentence starts here.'
            'nl | Sentence starts here.'
        :param fn: filename
        :param featObj: Object that contains the feature values to discern
        :return: list of lists representation of features and their truth values
        """

        def avgWordLength(lineTokens):
            # Compute avg word length
            length = 0
            for word in lineTokens:
                length += len(word)

            avgWordLen = length / len(lineTokens)  # Should always be 15
            return True if avgWordLen > 5 else False

        def doubleA(exampleChars):
            # Check for double 'aa'
            for i in range(1, len(exampleChars) - 1):
                if exampleChars[i - 1] == exampleChars[i]:
                    if exampleChars[i].lower() == 'a':
                        return True
            return False

        def doubleU(exampleChars):
            # Check for double 'uu'
            for i in range(1, len(exampleChars) - 1):
                if exampleChars[i - 1] == exampleChars[i]:
                    if exampleChars[i].lower() == 'u':
                        return True
            return False

        def doubleK(exampleChars):
            # Check for double 'kk'
            for i in range(1, len(exampleChars) - 1):
                if exampleChars[i - 1] == exampleChars[i]:
                    if exampleChars[i].lower() == 'k':
                        return True
            return False

        def totalDoubleCount(exampleChars):
            # Check for total double letter count > 2
            count = 0
            for i in range(1, len(exampleChars) - 1):
                if exampleChars[i - 1] == exampleChars[i]:
                        count += 1
            if count > 2:
                return True

            return False

        def containQ(exampleChars):
            # Contain the letter Q?
            for chr in exampleChars:
                if chr.lower() == 'q':
                    return True
            return False

        def containX(exampleChars):
            # Contain the letter X?
            for chr in exampleChars:
                if chr.lower() == 'x':
                    return True
            return False

        features = []
        with open(fn) as f:
            for line in f:
                if test == False: # If training
                    # Ignore incorrectly labeled data
                    if line[:2] != 'en' and line[:2] != 'nl':
                        continue
                    language = line[:2]
                    exampleFeatures = [0 for x in range(11)]
                    exampleString = line[5:]  # Truncate 'en | ' and \n
                else: # If testing
                    exampleFeatures = [0 for x in range(10)]
                    exampleString = line

                exampleChars = list(exampleString)
                lineTokens = exampleString.split(' ')
                exampleFeatures[0] = self.specialChars(exampleChars)
                exampleFeatures[1] = self.nlWords(lineTokens)
                exampleFeatures[2] = self.enWords(lineTokens)
                exampleFeatures[3] = doubleU(exampleChars)
                exampleFeatures[4] = doubleK(exampleChars)
                exampleFeatures[5] = totalDoubleCount(exampleChars)
                exampleFeatures[6] = avgWordLength(lineTokens)
                exampleFeatures[7] = doubleA(exampleChars)
                exampleFeatures[8] = containQ(exampleChars)
                exampleFeatures[9] = containX(exampleChars)
                if test == False:
                    exampleFeatures.append(language)
                features.append(exampleFeatures)
        f.close()

        return features