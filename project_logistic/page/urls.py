from django.urls import path
from . import views
from django.contrib.auth.views import LoginView
from .views import product_list

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('principal/', views.Principal, name='Principal'),
    path('addtemplate/', views.Addtemplate, name='Addtemplate'),
    path('sellers/', views.Sellers, name='Sellers'),
    path('products/', views.Products, name='Product'),
    path('usuario/',views.Prueba),
    path('admi/', views.Admin, name='Admin'),
    path('addtemplate/', views.Add, name='Add'),
    path('sells/', views.Sells, name='Sells'),
]