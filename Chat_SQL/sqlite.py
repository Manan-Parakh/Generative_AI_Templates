import sqlite3

# Connect to sqlite3
connection = sqlite3.Connection('student.db')

# Create a Cursor to create table and enter data
cursor = connection.cursor()

# Create the Table
table_info = """
create table STUDENT(NAME VARCHAR(25), CLASS VARCHAR(25), SECTION VARCHAR(25), MARKS INT)
"""

cursor.execute(table_info)

# Insert some records
cursor.execute("""Insert into STUDENT values('Krish',"Data Science","A",90)""")
cursor.execute("""Insert into STUDENT values('John','Data Science','B',100)""")
cursor.execute("""Insert into STUDENT values('Mukesh','Data Science','A',86)""")
cursor.execute("""Insert into STUDENT values('Jacob','Devops','A',50)""")
cursor.execute("""Insert into STUDENT values('Dipesh','Devops','A',35)""")

# Display all the records
print("The Inserted Records are:")
data = cursor.execute('SELECT * from STUDENT')

for row in data:
    print(row)

# Commit the changes to the database
connection.commit()
connection.close()