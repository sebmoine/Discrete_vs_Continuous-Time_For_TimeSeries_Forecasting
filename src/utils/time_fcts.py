import logging

def print_time(start,end):
    time = end - start

    days = time // (86400)
    hours = (time % 86400)// 3600
    mins = (time % 3600) // 60
    secs = time % 60
    if logging:
        print(f"{int(days)} day(s), {int(hours)} hour(s), {int(mins)} min(s) and {secs:.3f} sec(s).\n")
    else:
        print(f"{int(days)} day(s), {int(hours)} hour(s), {int(mins)} min(s) and {secs:.3f} sec(s).\n")