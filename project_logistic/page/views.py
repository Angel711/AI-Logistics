from django.shortcuts import render
from django.http import HttpResponse
from .models import Example

# Create your views here.

def Prueba(request):
    exame = Example(headling='este es un texto')
    exame.save()
    return HttpResponse('Datos guardados')