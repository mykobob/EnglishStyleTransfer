# Read King James Version into a dict of
# { chapter_name : { chapter_idx : verses_list_start_with_1 } }

from collections import defaultdict
import json
import re

def all_books():
    books_list = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
                 "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings",
                 "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah",
                 "Esther", "Job", "Psalm", "Proverbs", "Ecclesiastes",
                 "Song of Solomon", "Isaiah", "Jeremiah", "Lamentations", "Ezekiel",
                 "Daniel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah",
                 "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi",
                 "Matthew", "Mark", "Luke", "John", "Acts", "Romans",
                 "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
                 "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
                 "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James",
                 "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude",
                 "Revelation"]
    return books_list

def tokenize(verse_text):
    return re.findall(r"[\w']+|[\"-.,!?:;()]", verse_text)

def read_kjv(file_name):
    """
    Returns dict of entire King James Version from .csv file
    { chapter_name : { chapter_idx : verses_list_start_with_1 } }
    """

    books = all_books()
    kjv = {}
    for book in books:
        kjv[book] = defaultdict(defaultdict)
    print()

    # non_alphanumeric = {}
    # longest = 0
    with open(file_name) as kjv_lines:
        for line in kjv_lines:
            line = line.replace('"', "").strip().split(",", 4)
            this_book = books[int(line[1]) - 1]
            this_chapter = int(line[2])
            this_verse = int(line[3])
            verse_text = line[4]
            tokenized = tokenize(verse_text)
            kjv[this_book][this_chapter][this_verse] = tokenized
#             longest = max(longest, max([len(verse) for verse in kjv[this_book][this_chapter]]))
#             this_verse = len(kjv[this_book][this_chapter])
            # This inserts an index at chapter -1, so be careful using it to debug
            # if this_verse == 1:
            #     print("Starting next chapter")
            #     print(kjv[this_book][this_chapter-1])
            #     input()
            # print(kjv[this_book][this_chapter])

    # print('longest verse is', longest)
    return kjv

def read_esv(src_text):
    # Get Book with number of chapters and number of verses
    # dict of book name to array of numbers for # of verses
    longest = 0
    with open(src_text) as f:
        text = json.load(f)
        info = defaultdict(lambda: defaultdict(list))
        for verse_info in text:
            book_name = verse_info['book_name']
            chap_num = int(verse_info['chapter_id'])
            verse_text = verse_info['verse_text']

            tokenized = tokenize(verse_text)
            longest = max(longest, len(tokenized))
            info[book_name][chap_num].append(tokenized)
    print(f'longest esv is {longest}')
    return info

if __name__ == '__main__':
    kjv = read_kjv("data/kjv.csv")
    print(kjv["John"][3][16])
    
    esv = read_esv('data/esv.txt')
    print(esv["John"][3][2])
      