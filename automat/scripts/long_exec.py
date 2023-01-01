from datetime import datetime
import time 
# datetime object containing current date and time
# https://stackoverflow.com/questions/12919980/nohup-is-not-writing-log-to-output-file

while True:
	now=datetime.now()
	# dd/mm/YY H:M:S
	dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
	print("date and time =", dt_string)	
	time.sleep(30)
