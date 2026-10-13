# ee542_lab5

What the program does:
1. Connect to the ThingsBoard REST API.
2. Retrieve the latest location of each group member.
3. Retrieve the locations of the USC libraries.
4. Calculate the average (center) location of the group
5. Find the closest library to the group center.
6. Set a “recommended” attribute to True for the library chosen 
7. POST “recommended” attribute back to ThingsBoard.

## Running Instructions
```python3 nearest_library.py```

You will be prompted for the following: 

ThingsBoard email [tenant@thingsboard.org]:
ThingsBoard password:tenant
