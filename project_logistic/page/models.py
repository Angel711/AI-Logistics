from django.db import models

# Create your models here.

class Example(models.Model):
    headling = models.CharField(max_length=255)
    in_completed = models.CharField(max_length=200)
    name = models.CharField(max_length=100, null=True)

    def __str__(self):
        return self.headling


class User(models.Model):
    userName = models.CharField(max_length=50)
    lastName = models.CharField(max_length=50)
    email = models.EmailField()
    phone = models.CharField(max_length=20)
    password = models.CharField(max_length=50)
    role = models.CharField(max_length=50)
    status = models.CharField(max_length=50)

    def __str__(self):
        return self.userName

class Product(models.Model):
    productName = models.CharField(max_length=50)
    productPrice = models.IntegerField()
    productQuantity = models.IntegerField()
    productDescription = models.TextField()
    productImage = models.ImageField(upload_to='product_images/')
    productCategory = models.CharField(max_length=50)
    productStatus = models.CharField(max_length=50)
    productTendency = models.CharField(max_length=50)

    def __str__(self):
        return self.productName

