const questions = [];
let lastRequestDateTime = '2022/01/01 00:00:00'

// 質問取得
async function getQuestion() {
    const requestDate =  (new Date()).toLocaleString();
    const comments = await callApi().then(res => {return res.json()}).catch(err => {alert(err)});
    lastRequestDateTime = requestDate;
    console.log(comments['convs']);
    comments['convs'].forEach(comment => {
        questions.push(comment);
    });
}

// 質問送信
function setQuestion(messgae_id, question, answer) {
    let template = document.querySelector('#message-template');
    let newMessage = template.content.cloneNode(true);

    newMessage.querySelector('article').dataset.messageId = messgae_id;
    newMessage.querySelector('.question').textContent = question;
    newMessage.querySelector('.answer').textContent = answer;

    const message_area = document.getElementById('message-card-area');
    addScrollEvent(newMessage.querySelector('article'));
    message_area.append(newMessage);
}

// スクロールイベント
function addScrollEvent(endroll){
    // let distance = window.innerHeight + endroll.clientHeight
    // const duration = distance / 100;
    // const messages = document.querySelectorAll('.message');
    // const messages_count = messages.length - 1;
    // console.log(messages)
    // console.log(messages[messages_count].clientHeight)
    // distance = distance + (messages_count * 100);
    // console.log(distance)
    // endroll.style.top = distance + 'px';

    // endroll.style.animationDuration = `${duration}s`
    endroll.style.animationDuration = '15s';
    endroll.classList.add('never-enough');
    endroll.addEventListener('animationend', () => {
        endroll.remove();
    });
}

// Fetch API
async function callApi() {
    return await fetch('http://localhost:8080/api/v1/comments?llt=' + encodeURIComponent(lastRequestDateTime))
}

// コメント取得
setInterval(() => {
    console.log('fetch start');
    getQuestion();
    console.log('fetch end');
}, 3000);

// コメント送信
setInterval(function() {
    let comment = questions.shift();
    if (comment) {
        setQuestion('', comment['usr'], comment['sys']);
    }
}, 3000);
