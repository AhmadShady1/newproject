var productName = document.getElementById("productName");
var productPrice = document.getElementById("productPrice");
var productCatgory = document.getElementById("productCatgory");
var productDesc = document.getElementById("productDesc");

var productContiner = [];

function addProduct() {
    var product = {
        name:productName.value,
        price:productPrice.value,
        Catgory:productCatgory.value,
        Desc:productDesc.value,



    }
    productContiner.push(product)
    console.log(productContiner)
    clear();

}
function clear() {
    productName.value ="";
    productPrice.value = "";
    productCatgory.value = "";
    productDesc.value = "";

}
   
      
  

