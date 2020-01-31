from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import JSONParser
import joblib


def get(self, request):
        if request.method == 'GET':

            # sentence is the query we want to get the prediction for
            params = request.GET.get('predict')


            model = joblib.load("./model/SVC.joblib")
            # predict method used to get the prediction
            response = model.predict(params)

            # returning JSON response
            return JsonResponse(response)
