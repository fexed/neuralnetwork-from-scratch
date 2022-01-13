def update_progress(progress, barlength=90, suffix="", fill="#"):
    num = int(round(barlength*progress))
    txt = "\r" + suffix + " [" + "#"*num + "-"*(barlength - num) + "] " + "{:.2f}".format(progress*100) + "%"
    print(txt, end="")
