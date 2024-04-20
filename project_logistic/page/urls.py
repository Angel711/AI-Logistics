from django.urls import path
from . import views
from django.contrib.auth.views import LoginView
from .views import product_list

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('principal/', views.Principal, name='Principal'),
    path('card/', views.Card, name='Card'),
    path('addtemplate/', views.Addtemplate, name='Addtemplate'),
    path('sellers/', views.Sellers, name='Sellers'),
    path('products/', product_list, name='product_list'),
    path('usuario/',views.Prueba)
]