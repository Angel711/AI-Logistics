from django.shortcuts import render
from django.http import HttpResponse
from .models import Product, User
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
#El modelo User de Django tiene un conjunto específico de campos que puedes utilizar. Los campos básicos son username, first_name, last_name, email, password, groups, user_permissions, is_staff, is_active, is_superuser, last_login, y date_joined

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

def Prueba(request):
    user = User(username='Juan789', first_name='Juan', last_name='Perez', email='user@example.com')
    user.set_password('password')
    user.save()
    return HttpResponse('datos guardados')

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return render(request, 'principal.html', {})
        else:
            return HttpResponse('Invalid login: {}'.format(str(user)))

    return render(request, 'login.html')

