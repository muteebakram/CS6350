variety = ["Alphonso", "Keitt", "Haden"]
color = ["Red", "Yellow", "Green"]
smell = ["Sweet", "None"]
time = ["One", "Two"]
ripe = ["False", "True"]

# for v in variety:
#     for c in color:
#         for s in smell:
#             for t in time:
#                 for r in ripe:
#                     print(f"{v:<10} {c:<10} {s:<10} {t:<10} {r:<10}")


for v in variety:
    for c in color:
        for s in smell:
            for t in time:
                for r in ripe:
                    c1 = v == "Alphonso" and c == "Red" and s == "None" and t == "Two" and r == "False"
                    c2 = v == "Keitt" and c == "Red" and s == "None" and t == "One" and r == "True"
                    c3 = v == "Alphonso" and c == "Yellow" and s == "Sweet" and t == "Two" and r == "True"
                    c4 = v == "Keitt" and c == "Green" and s == "None" and t == "Two" and r == "False"
                    c5 = v == "Haden" and c == "Green" and s == "Sweet" and t == "One" and r == "True"
                    c6 = v == "Alphonso" and c == "Yellow" and s == "None" and t == "Two" and r == "False"
                    c7 = v == "Keitt" and c == "Yellow" and s == "Sweet" and t == "One" and r == "False"
                    c8 = v == "Alphonso" and c == "Red" and s == "Sweet" and t == "Two" and r == "True"

                    if c1 or c2 or c3 or c4 or c5 or c6 or c7 or c8:
                        print(f"{v:<10} {c:<10} {s:<10} {r:<10}")
