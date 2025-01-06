from flask import redirect,  url_for, render_template,request,Flask,Response,json,send_file
import docx2txt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy_try import spacy_match
import pandas as pd
import nltk
import io
import random
import jinja2
#from pdf2docx import Converter
from bs4 import BeautifulSoup
from datetime import datetime
from bills import parser
import openpyxl
from io import BytesIO
from excel_edit import edit_excel, read_excel


env = jinja2.Environment()
env.globals.update(zip=zip)

app = Flask(__name__)

df_for_excel = pd.DataFrame(columns=['Name','Match_Percentage'])
df_for_excel = pd.DataFrame(None)
#df_for_excel = df_for_excel.reset_index(drop=True, inplace=True)

count_excel_path = 'count.xlsx'

@app.route('/',methods=['GET','POST'])

@app.route('/jdres/',methods=['GET','POST'])

def home():
	#getting initial count of resumes checked
	initial_count = read_excel(count_excel_path,'Sheet1','A1')
	df_for_excel = pd.DataFrame(None)

	return render_template('index.html',initial_count=initial_count)


@app.route("/newresult",methods=['GET','POST'])

def newresult():
	if request.method == 'POST':
		#jd = request.files['jd']
		#resume = request.files['resume']
		if request.files['jd'].filename != "":
			jd = request.files['jd']
			if request.files['resume'].filename!="":
				resume = request.files.getlist('resume')
				check_type = request.form['role_type']

				name_list = []

				i = 0
				while i < len(resume):
					name_list.append(resume[i].filename.split(".")[0])
					i=i+1
				#counting how many resumes are being checked
				count_of_resume = len(name_list)
				print("the count of resume is:::")
				print(count_of_resume)
				#getting initial count of resumes checked
				initial_count = read_excel('count.xlsx','Sheet1','A1')
				print("the intial count value is::::")
				print(initial_count)
				#adding new resume count values
				new_count_value = initial_count+count_of_resume
				edit_excel('count.xlsx','Sheet1','A1',new_count_value)

				id_list = []
				percentage,word_list,common_list,label_x,resume_val,jd_val,summary=spacy_match(resume,jd,check_type)				
				k=0
				while k < len(percentage):
					id_list.append(k)
					k=k+1
				zip_list_1 = zip(label_x,resume_val,jd_val)
				zip_list = zip(percentage,word_list,common_list,name_list,id_list)
				
				df_for_excel["Name"] = pd.Series(name_list)
				df_for_excel["Match_Percentage"] = pd.Series(percentage)
				
				
				return render_template("newresult.html",zip_list = zip_list,zip_list_1=zip_list_1, summary=summary)
			else:
				return render_template("notfound.html")
		else: 
			return render_template("notfound.html")

@app.route('/downloadresult/', methods=['GET', 'POST'])
def downloadresult():
	if request.method == 'POST':
		output = BytesIO()
		writer = pd.ExcelWriter(output, engine='xlsxwriter')
		df_for_excel.to_excel(writer,startrow = 0, merge_cells = False, sheet_name = "Sheet_1")
		
		workbook = writer.book
		worksheet = writer.sheets["Sheet_1"]
		writer.close()
		output.seek(0)
		
		return send_file(output, download_name="output.xlsx", as_attachment=True)




if __name__ == '__main__':
    app.debug = True
    app.run(host = '0.0.0.0', port= 8080)