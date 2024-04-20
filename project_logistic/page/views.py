from django.shortcuts import render
from django.http import HttpResponse
from .models import Example
from .models import Product

def product_list(request):
    products = Product.objects.all()
    return render(request, 'tarjeta.html', {'products': products})

# Create your views here.
def Login(request):
    return render(request, 'login.html', {})

def Principal(request):
    return render(request, 'principal.html', {})

def Card(request):
    return render(request, 'card.html', {})

def Addtemplate(request):
    return render(request, 'addtemplate.html', {})

def Sellers(request):
    return render(request, 'sellers.html', {})