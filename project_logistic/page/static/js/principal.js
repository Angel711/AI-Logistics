var
    $btnViewAddTemplate = ("#btnViewAddTemplate");


$btnViewAddTemplate.on("click", function() {
    window.location.href = "{% include 'addtemplate.html' %}";
});