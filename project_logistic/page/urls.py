from django.urls import path
from . import views
from .views import product_list

urlpatterns = [
    path('products/', product_list, name='product_list'),
]

urlpatterns = [
    path('login/', views.Login, name='Login'),
    path('principal/', views.Principal, name='Principal'),
    path('card/', views.Card, name='Card'),
    path('addtemplate/', views.Addtemplate, name='Addtemplate'),
    path('sellers/', views.Sellers, name='Sellers')
]