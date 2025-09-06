if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    # Log de inicio
    print(f"🚀 Iniciando servidor en puerto {port}")
    print(f"🔗 Health check disponible en: http://0.0.0.0:{port}/")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"❌ Error iniciando servidor: {e}")