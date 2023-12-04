######################          Import packages      ###################################
from flask import Blueprint, render_template, flash, url_for, request
from flask_login import login_required, current_user
from __init__ import create_app, db
import pickle
import numpy as np

model = pickle.load(open('model1.pkl', 'rb'))

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/profile') # profile page that return 'profile'
@login_required
def profile():
    return render_template('profile.html', name=current_user.name)

@main.route('/indexc')
@login_required
def apti():
    return render_template('indexc.html', name=current_user.name)

@main.route('/brdigi')
def brdigi():
    return render_template('DigitalMarketing.html', name=current_user.name)

@main.route('/coml')
def coml():
    return render_template('ML.html', name=current_user.name)

@main.route('/cowebdev')
def cowebdev():
    return render_template('WebDev.html', name=current_user.name)

@main.route('/copyth')
def copyth():
    return render_template('PythonForBeg.html', name=current_user.name)

@main.route('/cojava')
def cojava():
    return render_template('JavaForBeg.html', name=current_user.name)

@main.route('/prof')
def prof():
    return render_template('professions.html', name=current_user.name)

@main.route('/proeng')
def proeng():
    return render_template('engineer.html', name=current_user.name)

@main.route('/proarch')
def proarch():
    return render_template('Architect.html', name=current_user.name)

@main.route('/proscient')
def proscient():
    return render_template('Scientist.html', name=current_user.name)

@main.route('/proteach')
def proteach():
    return render_template('Teacher.html', name=current_user.name)

@main.route('/prodoc')
def prodoc():
    return render_template('Doctor.html', name=current_user.name)

@main.route('/rdvoc')
def rdvoc():
    return render_template('rdvoc.html', name=current_user.name)

@main.route('/rdvocdm')
def rdvocdm():
    return render_template('digitalBus.html', name=current_user.name)

@main.route('/rditi')
def rditi():
    return render_template('rditi.html', name=current_user.name)

@main.route('/rdiploma')
def rdiploma():
    return render_template('rdiploma.html', name=current_user.name)

@main.route('/rdparamed')
def rdparamed():
    return render_template('rdparamed.html', name=current_user.name)

@main.route('/rdpoly')
def rdpoly():
    return render_template('rdpoly.html', name=current_user.name)

@main.route('/rdmp')
def rdmp():
    return render_template('Roadmap.html', name=current_user.name)

@main.route('/apt_science')
def aptsci():
    return render_template('index_sci.html', name=current_user.name)

@main.route('/apt_commerce')
def aptcom():
    return render_template('index_com.html', name=current_user.name)

@main.route('/apt_arts')
def aptart():
    return render_template('index_art.html', name=current_user.name)

@main.route('/index_voc')
def vocmeth():
    return render_template('index_voc.html', name=current_user.name)

@main.route('/',methods=['GET'])
def newpage():
    return render_template('firstc.html')

@main.route('/300')
def Arts():
    return render_template('300.html')

@main.route('/200')
def Commerce():
    return render_template('200.html')

@main.route('/100')
def Science():
    return render_template('100.html')

@main.route('/PA5')
def PA5():
    return render_template('PA5.html')

@main.route('/indexc',methods=['POST'])
def helloworld():
    return render_template('indexc.html')

@main.route('/resultc',methods=['POST'])
def insurance():
    sq1=request.form['sq1']
    sq2=request.form['sq2']
    sq3=request.form['sq3']
    sq4=request.form['sq4']
    sq5=request.form['sq5']
    sq6=request.form['sq6']
    sq7=request.form['sq7']
    sq8=request.form['sq8']
    sq9=request.form['sq9']
    sq10=request.form['sq10']

    cq1=request.form['cq1']
    cq2=request.form['cq2']
    cq3=request.form['cq3']
    cq4=request.form['cq4']
    cq5=request.form['cq5']
    cq6=request.form['cq6']
    cq7=request.form['cq7']
    cq8=request.form['cq8']
    cq9=request.form['cq9']
    cq10=request.form['cq10']

    aq1=request.form['aq1']
    aq2=request.form['aq2']
    aq3=request.form['aq3']
    aq4=request.form['aq4']
    aq5=request.form['aq5']
    aq6=request.form['aq6']
    aq7=request.form['aq7']
    aq8=request.form['aq8']
    aq9=request.form['aq9']
    aq10=request.form['aq10']


    arr = np.array([[sq1,sq2,sq3,sq4,sq5,sq6,sq7,sq8,sq9,sq10,cq1,cq2,cq3,cq4,cq5,cq6,cq7,cq8,cq9,cq10,aq1,aq2,aq3,aq4,aq5,aq6,aq7,aq8,aq9,aq10]])
    pred = model.predict(arr)
    return render_template('resultc.html', data=pred)


######################################### Arts 2 QUIZ ###################################################

@main.route('/index_art')
def helloworld_a():
    return render_template('index_art.html')

modela = pickle.load(open('model_art.pkl', 'rb'))
@main.route('/result_art', methods=['POST'])
def artmeth():
    sq1=request.form['sq1']
    sq2=request.form['sq2']
    sq3=request.form['sq3']
    sq4=request.form['sq4']
    sq5=request.form['sq5']
    sq6=request.form['sq6']
    sq7=request.form['sq7']

    cq1=request.form['cq1']
    cq2=request.form['cq2']
    cq3=request.form['cq3']
    cq4=request.form['cq4']
    cq5=request.form['cq5']
    cq6=request.form['cq6']
    cq7=request.form['cq7']

    aq1=request.form['aq1']
    aq2=request.form['aq2']
    aq3=request.form['aq3']
    aq4=request.form['aq4']
    aq5=request.form['aq5']
    aq6=request.form['aq6']
    aq7=request.form['aq7']

    arr = np.array([[sq1,sq2,sq3,sq4,sq5,sq6,sq7,cq1,cq2,cq3,cq4,cq5,cq6,cq7,aq1,aq2,aq3,aq4,aq5,aq6,aq7]])
    preda = modela.predict(arr)
    return render_template('result_art.html', data=preda)

@main.route('/305')
def ArtsThree():
    return render_template('305.html')
@main.route('/350')
def ArtsTwo():
    return render_template('350.html')
@main.route('/395')
def ArtsOne():
    return render_template('395.html')

################################COMMERCE 2 QUIZ ##########################################
@main.route('/index_com')
def helloworld_c():
    return render_template('index_com.html')

modelc = pickle.load(open('model_com.pkl', 'rb'))
@main.route('/result_com', methods=['POST'])

def commeth():
    mq1=request.form['mq1']
    mq2=request.form['mq2']
    mq3=request.form['mq3']
    mq4=request.form['mq4']
    mq5=request.form['mq5']

    jq1=request.form['jq1']
    jq2=request.form['jq2']
    jq3=request.form['jq3']
    jq4=request.form['jq4']
    jq5=request.form['jq5']

    bq1=request.form['bq1']
    bq2=request.form['bq2']
    bq3=request.form['bq3']
    bq4=request.form['bq4']
    bq5=request.form['bq5']

    lq1=request.form['lq1']
    lq2=request.form['lq2']
    lq3=request.form['lq3']
    lq4=request.form['lq4']
    lq5=request.form['lq5']

    caq1=request.form['caq1']
    caq2=request.form['caq2']
    caq3=request.form['caq3']
    caq4=request.form['caq4']
    caq5=request.form['caq5']


    arr = np.array([[mq1,mq2,mq3,mq4,mq5,jq1,jq2,jq3,jq4,jq5,bq1,bq2,bq3,bq4,bq5,lq1,lq2,lq3,lq4,lq5,caq1,caq2,caq3,caq4,caq5]])
    predc = modelc.predict(arr)
    return render_template('result_com.html', data=predc)

@main.route('/210')
def ComOne():
    return render_template('210.html')

@main.route('/230')
def ComTwo():
    return render_template('230.html')

@main.route('/250')
def ComThree():
    return render_template('250.html')

@main.route('/270')
def ComFour():
    return render_template('270.html')

@main.route('/290')
def ComFive():
    return render_template('290.html')

################################SCIENCE 2 QUIZ ##########################################
@main.route('/index_sci')
def helloworld_s():
    return render_template('index_sci.html')

models = pickle.load(open('model_sci.pkl', 'rb'))
@main.route('/result_sci', methods=['POST'])
def scimeth():
    mq1=request.form['mq1']
    mq2=request.form['mq2']
    mq3=request.form['mq3']
    mq4=request.form['mq4']
    mq5=request.form['mq5']

    eq1=request.form['eq1']
    eq2=request.form['eq2']
    eq3=request.form['eq3']
    eq4=request.form['eq4']
    eq5=request.form['eq5']

    arq1=request.form['arq1']
    arq2=request.form['arq2']
    arq3=request.form['arq3']
    arq4=request.form['arq4']
    arq5=request.form['arq5']

    aeq1=request.form['aeq1']
    aeq2=request.form['aeq2']
    aeq3=request.form['aeq3']
    aeq4=request.form['aeq4']
    aeq5=request.form['aeq5']

    pq1=request.form['pq1']
    pq2=request.form['pq2']
    pq3=request.form['pq3']
    pq4=request.form['pq4']
    pq5=request.form['pq5']


    arr = np.array([[mq1,mq2,mq3,mq4,mq5,eq1,eq2,eq3,eq4,eq5,arq1,arq2,arq3,arq4,arq5,aeq1,aeq2,aeq3,aeq4,aeq5,pq1,pq2,pq3,pq4,pq5]])
    preds = models.predict(arr)
    return render_template('result_sci.html', data=preds)

@main.route('/110')
def SciOne():
    return render_template('110.html')

@main.route('/130')
def SciTwo():
    return render_template('130.html')

@main.route('/150')
def SciThree():
    return render_template('150.html')

@main.route('/170')
def SciFour():
    return render_template('170.html')

@main.route('/190')
def SciFive():
    return render_template('190.html')
#

modelv = pickle.load(open('model_voc.pkl', 'rb'))
@main.route('/result_voc', methods=['POST'])
def vocmeth2():
    sq1=request.form['sq1']
    sq2=request.form['sq2']
    sq3=request.form['sq3']
    sq4=request.form['sq4']
    sq5=request.form['sq5']

    cq1=request.form['cq1']
    cq2=request.form['cq2']
    cq3=request.form['cq3']
    cq4=request.form['cq4']
    cq5=request.form['cq5']

    aq1=request.form['aq1']
    aq2=request.form['aq2']
    aq3=request.form['aq3']
    aq4=request.form['aq4']
    aq5=request.form['aq5']

    arr = np.array([[sq1,sq2,sq3,sq4,sq5,cq1,cq2,cq3,cq4,cq5,aq1,aq2,aq3,aq4,aq5]])
    predv = modelv.predict(arr)
    return render_template('result_voc.html', data=predv)

@main.route('/1800')
def VocThree():
    return render_template('1800.html')
@main.route('/1200')
def VcTwo():
    return render_template('1200.html')
@main.route('/600')
def VocOne():
    return render_template('600.html')
@main.route('/456')
def VocFour():
    return render_template('456.html')

#

modeld = pickle.load(open('model_voc.pkl', 'rb'))
@main.route('/result_voc', methods=['POST'])
def dipmeth():
    sq1=request.form['sq1']
    sq2=request.form['sq2']
    sq3=request.form['sq3']
    sq4=request.form['sq4']
    sq5=request.form['sq5']

    cq1=request.form['cq1']
    cq2=request.form['cq2']
    cq3=request.form['cq3']
    cq4=request.form['cq4']
    cq5=request.form['cq5']

    aq1=request.form['aq1']
    aq2=request.form['aq2']
    aq3=request.form['aq3']
    aq4=request.form['aq4']
    aq5=request.form['aq5']

    arr = np.array([[sq1,sq2,sq3,sq4,sq5,cq1,cq2,cq3,cq4,cq5,aq1,aq2,aq3,aq4,aq5]])
    predd = modeld.predict(arr)
    return render_template('result_voc.html', data=predd)

@main.route('/1800')
def DipThree():
    return render_template('1800.html')
@main.route('/1200')
def DipTwo():
    return render_template('1200.html')
@main.route('/600')
def DipOne():
    return render_template('600.html')



app = create_app() # we initialize our app using the __init__.py function
if __name__ == '__main__':
    with app.app_context():
        db.create_all() # create the SQLite database
    app.run(debug=True) # run the app on debug mode
