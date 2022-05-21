from bottle import *

@route("/")
def login():
    return template("sample")

@route('/assets/<filename>')
def route_css(filename):
    return static_file(filename, root='assets/')

@error(404)
def error404(error):
    return '<font size=6><left> You got lost?? <br><br/>\
    Here is the login URL ->> </left></font>\
    <a href = "http://localhost:8080/">Click!!</a>'.format(error=error)

run(host='localhost',port=8080,debug=True)
