from django.shortcuts import render
from django.http import HttpResponse
from .models import Example
from django.shortcuts import render
from .models import Product

def product_list(request):
    products = Product.objects.all()
    return render(request, 'tarjeta.html', {'products': products})

# Create your views here.

def Prueba(request):
    exame = Example(headling='este es un texto')
    exame.save()
    return HttpResponse('Datos guardados')