from bottle import *
import exract_comments

@route("/")
def top():
    return template("sample")

@route('/assets/<filename>')
def route_css(filename):
    return static_file(filename, root='assets/')

@route('/api/v1/comments')
def get_all_comments():
    response.headers['Content-Type'] = 'application/json'
    response.headers['Cache-Control'] = 'no-cache'

    llt = request.query.decode()['llt'].replace('/', '-')
    return exract_comments.extract_convs_to_json(llt)

@error(404)
def error404(error):
    return '<font size=6><left> You got lost?? <br><br/>\
    Here is the login URL ->> </left></font>\
    <a href = "http://localhost:8080/">Click!!</a>'.format(error=error)

run(host='localhost', port=8080, debug=True, reloader=True)
