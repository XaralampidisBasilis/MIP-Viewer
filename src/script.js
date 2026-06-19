
import { setTensorflow } from './javascript/tensorflow';
import Experience from './javascript/Experience'

(async () => 
{
    await setTensorflow()

    const canvas = document.querySelector('canvas.webgl')
    const experience = new Experience(canvas) 
}
)()
