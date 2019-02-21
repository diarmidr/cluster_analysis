import pandas as pd
data = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
day=0
calendar_view = pd.DataFrame()
while day < len(data):
    print(day)
    counter = 0
    remaining_data = data[day:]
    week_list = []
    while counter <= 6:

        week_list= week_list + [remaining_data[counter]]
        counter = counter + 1
    calendar_view = calendar_view.append([week_list])
    day = day + counter

print(calendar_view)

