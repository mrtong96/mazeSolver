var express = require('express');
var app = express();
var port = process.env.PORT || 80;
var multer = require('multer');

var storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/');
  },
  filename: function (req, file, cb) {
    cb(null, file.fieldname + Date.now()+".png");
  }
})

var upload = multer({dest: 'uploads/', storage: storage});
var spawn = require('child_process').spawn;
var path = require('path');
var rootPath = path.join(__dirname);
app.use(express.static(path.join(rootPath + '/results')));
app.use(express.static(path.join(rootPath + '/uploads')));

/*app.use(multer({ dest: './uploads/',
    rename: function (fieldname, filename) {
        return filename+Date.now();
    },
    onFileUploadStart: function (file) {
        console.log(file.originalname + ' is starting ...');
    },
    onFileUploadComplete: function (file) {
        console.log(file.fieldname + ' uploaded to  ' + file.path)
    }
}));*/
//View Setup
//app.use(express.static(path.join(rootPath + '/public'))); // public
//app.use('/angular', express.static(path.join(rootPath + '/angular')));
//app.set('view engine', 'ejs');
// pyshit ======================================================================
/*py.stdout.on('data', function(data){
    console.log(data.toString());
});

py.stdin.write(JSON.stringify('lol'));
py.stdin.end();*/

// routes ======================================================================
app.get('/',function(req,res){
      res.sendFile(__dirname + "/index.html");
});

app.post('/api/maze', upload.single('maze'), function(req, res){
    var py = spawn('python', ['maze_solver.py']);
    var finalPath = "";
    py.stdout.on('data', function(data){
        finalPath = data.toString('utf8');
    });
    py.stdout.on('end', function(){
        res.send("<img src=/" + req.file.filename + "><img src=/" + finalPath + ">");
    });
    py.stdin.write(JSON.stringify(req.file.filename));
    py.stdin.end();	
});

app.listen(port, function(){
	console.log('Listening at http://localhost:%s',port);
});
