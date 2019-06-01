# Author Toshihiko Aoki
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import unicodedata
import six


class Pipeline(object):
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def __call__(self, text):
        for p in self.preprocessors:
            text = p(text)
        return text

    def append(self, preprocessor):
        self.preprocessors.append(preprocessor)

    def __str__(self):
        format_string = self.__class__.__name__ + '['
        for p in self.preprocessors:
            format_string += '\n'
            format_string += '    {0}'.format(p)
        format_string += '\n]'
        return format_string

    def __delitem__(self, key):
        for p in self.preprocessors:
            if str(p) == key:
                self.preprocessors.remove(p)


try:
    from mojimoji import han_to_zen


    class ToZenkaku(object):
        def __call__(self, text):
            return han_to_zen(text)

        def __str__(self):
            return self.__class__.__name__

except ImportError:
    # only alphabet and number
    UPPER = dict((0x0041 + ch, 0xFF21 + ch) for ch in range(26))
    LOWER = dict((0x0061 + ch, 0xFF41 + ch) for ch in range(26))
    NUMBER = dict((0x0030 + ch, 0xFF10 + ch) for ch in range(10))
    half_to_full = {**UPPER, **LOWER, **NUMBER}


    class ToZenkaku(object):
        def __call__(self, text):
            return text.translate(half_to_full)

        def __str__(self):
            return self.__class__.__name__


class ToUnicode(object):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""

    def __call__(self, text):
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python 3")

    def __str__(self):
        return self.__class__.__name__


class Normalize(object):

    def __init__(self, form="NFKC"):
        self.form = form

    def __call__(self, text):
        return unicodedata.normalize(self.form, text)

    def __str__(self):
        return self.__class__.__name__


class LowerCase(object):

    def __call__(self, text):
        return text.lower()

    def __str__(self):
        return self.__class__.__name__


class ReplaceNumber(object):

    def __init__(self, replacement='0'):
        self.replacement = replacement

    def __call__(self, text):
        return re.sub(r'\d+', self.replacement, text)

    def __str__(self):
        return self.__class__.__name__


class ReplaceURI(object):

    def __init__(self, replacement='link'):
        self.replacement = replacement

    def __call__(self, text):
        return re.sub(
            r'http\S+',
            self.replacement,
            text,
            flags=re.MULTILINE
        )

    def __str__(self):
        return self.__class__.__name__
