var express = require('express');
var app = express();
var port = process.env.PORT || 8080;
var multer = require('multer');
var multer = require('multer');
var upload = multer({dest: 'uploads/'});
var spawn = require('child_process').spawn;

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

app.post('/api/photo', upload.single('maze'), function(req, res){
    var py = spawn('python', ['maze_solver.py']);
    py.stdout.on('data', function(data){
        console.log(data.toString());
    });

    py.stdin.write(JSON.stringify(req.file.filename));
    py.stdin.end();
    //console.log(req.body);
    /*upload(req,res,function(err) {
        if(err) {
            return res.end("Error uploading file.");
        }
        res.end("File is uploaded");
    });*/
	res.end("File is uploaded");
});

app.listen(port, function(){
	console.log('Listening at http://localhost:%s',port);
});
