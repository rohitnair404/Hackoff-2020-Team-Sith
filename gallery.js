//var path = "enter the folder path where the images are stored, it should be wrt the gallery.html file";
var path="\frames_without_mask\"
var frames=['frame0','frame30','frame60','frame90','frame120','frame150','frame180','frame210','frame240','frame270','frame300','frame330','frame360','frame390','frame420','frame450','frame480'];
var pths=[];
for(var i=0;i<15;i++)
{
    pths.push(path+frames[i]+'.jpg');
}
var imgs=document.querySelectorAll(".imgg");
for(var i=0;i<15;i++)
{
    imgs[i].src=pths[i];
}

