def file_line_count(filepath):
    cnt = 0
    with open(filepath) as file:
        for line in file:
            cnt += 1
    return cnt
