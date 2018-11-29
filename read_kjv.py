# Read King James Version into a dict of
# { chapter_name : { chapter_idx : verses_list_start_with_1 } }

from collections import defaultdict


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


def read_kjv(file_name):
    """
    Returns dict of entire King James Version from .csv file
    { chapter_name : { chapter_idx : verses_list_start_with_1 } }
    """

    books = all_books()
    kjv = {}
    for book in books:
        kjv[book] = defaultdict(list)
    print()

    with open(file_name) as kjv_lines:
        for line in kjv_lines:
            line = line.replace('"', "").strip().split(",")
            this_book = books[int(line[1]) - 1]
            # print("Current book", this_book)
            this_chapter = int(line[2])
            # print("Current chapter", this_chapter)
            kjv[this_book][this_chapter].append(",".join(line[4:]))
            this_verse = len(kjv[this_book][this_chapter])
            # This inserts an index at chapter -1, so be careful using it to debug
            # if this_verse == 1:
            #     print("Starting next chapter")
            #     print(kjv[this_book][this_chapter-1])
            #     input()
            # print(kjv[this_book][this_chapter])

    return kjv


if __name__ == '__main__':
    kjv = read_kjv("data/kjv.csv")
    print(kjv["John"][3][15])

